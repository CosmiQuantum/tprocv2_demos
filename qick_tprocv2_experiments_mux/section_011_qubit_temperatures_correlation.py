import matplotlib.pyplot as plt
import numpy as np
import visdom
import math
from scipy.optimize import curve_fit
#import datetime                                                                                                                                                                                
from build_task import *
from build_state import *
from expt_config import *
from system_config import *
#from qick.asm_v2 import Sweep                                                                                                                                                                  
import logging
from expt_config import *
import copy
# import visdom                                                                                                                                                                                 
from scipy.signal import argrelextrema

class Temps_CorrelationExperiment:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder, round_num, signal, save_figs,
                 experiment = None,live_plot = None, fit_data = None, increase_qubit_reps = False,
                 qubit_to_increase_reps_for = None,multiply_qubit_reps_by = 0, verbose = False,
                 logger = None, qick_verbose=True, save_shots=False,set_relax_delay=False, relax_delay=1000,
                 unmasking_resgain = False, adjust_reps_to = None):

        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        #self.expt_name = "T1_ge"                                                                                                                                                               
        self.expt_name = "correlation_method"
        self.fit_data = fit_data
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.live_plot = live_plot
        self.signal = signal
        self.save_figs = save_figs
        self.verbose = verbose
        self.save_shots = save_shots
        self.set_relax_delay = set_relax_delay
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        ### Temp Bipass

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]
        if adjust_reps_to is not None:
            self.exp_cfg["reps"] = adjust_reps_to
        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            #print("___________")                                                                                                                                                               
            #print(expt_cfg)                                                                                                                                                                    

            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            #print(self.exp_cfg)                                                                                                                                                                
            #crash=crasher                                                                                                                                                                      
            if increase_qubit_reps:
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *= multiply_qubit_reps_by
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            if self.set_relax_delay:
                self.config['relax_delay'] = relax_delay
                print(f'set t1 relax delay to {relax_delay} us')




            #def __init__(self, QubitIndex, number_of_qubits, list_of_all_qubits,  outerFolder, round_num, signal, save_figs, experiment = None, live_plot = None,                              
     #            increase_qubit_reps = False, qubit_to_increase_reps_for = None, multiply_qubit_reps_by = 0, unmasking_resgain = False):                                                       
        #self.QubitIndex = QubitIndex                                                                                                                                                           
        #self.number_of_qubits = number_of_qubits                                                                                                                                               
        #self.outerFolder = outerFolder                                                                                                                                                         
        #self.expt_name = "correlation_method"                                                                                                                                                  
        #self.Qubit = 'Q' + str(self.QubitIndex)                                                                                                                                                
        #self.exp_cfg = expt_cfg[self.expt_name]                                                                                                                                                
        #self.round_num = round_num                                                                                                                                                             
        #self.live_plot = live_plot                                                                                                                                                             
        #self.signal = signal                                                                                                                                                                   
        #self.save_figs = save_figs                                                                                                                                                             
        #self.experiment = experiment                                                                                                                                                           
        #self.list_of_all_qubits = list_of_all_qubits                                                                                                                                           

        #if unmasking_resgain:                                                                                                                                                                  
        #    self.exp_cfg["list_of_all_qubits"] = [QubitIndex]                                                                                                                                  


        #if experiment is not None:                                                                                                                                                             
        #    self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)                                                                                                            
        #    self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)                                                                                                     
        #    self.config = {**self.q_config[self.Qubit], **self.exp_cfg}                                                                                                                        
        #    if increase_qubit_reps:                                                                                                                                                            
        #            if self.QubitIndex==qubit_to_increase_reps_for:                                                                                                                            
        #                print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")                                                                                           
        #                self.config["reps"] *= multiply_qubit_reps_by                                                                                                                          
        #    print(f'Q {self.QubitIndex + 1} Round {self.round_num} EF Rabi configuration: ', self.config)                                                                                      
        #    #self.config['relax_delay'] = relax_delay                                                                                                                                          
        #    #print(f'set t1 relax delay to {relax_delay} us')                                                                                                                                  


    def run(self, soccfg, soc):
        #print(self.config)                                                                                                                                                                     
        now = datetime.datetime.now()
        correlation = CorrelationGround(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay2'], cfg=self.config)
        if self.live_plot:
            tempvalue = 1
            #I1, Q1, gains1 = self.live_plotting(correlation, soc)                                                                                                                              
        else:
            iq_list1_rough = correlation.acquire(self.experiment.soc, progress=True)
            iq_list1_rough = np.array(iq_list1_rough)
            iq_list1_rough = iq_list1_rough.squeeze()
            iq_list1 = iq_list1_rough.transpose(0,2,1)

            I1_g = iq_list1[0,0]
            Q1_g = iq_list1[0,1]
            I2_g = iq_list1[1,0]
            Q2_g = iq_list1[1,1]

            #crash1 = crasher1                                                                                                                                                                  

        correlation_e = CorrelationExcited(soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'],
                                          cfg=self.config)
        if self.live_plot:
            tempvalue = 2
            #I2, Q2, gains2 = self.live_plotting(amp_rabi2, soc)                                                                                                                                
        else:
            #iq_list2 = correlation_e.acquire(soc, progress=True, readout_per_experiment=2, save_experiment=True)                                                                               
            iq_list2 = correlation_e.acquire(soc, progress=True)
            iq_list2 = np.array(iq_list2)
            iq_list2 = iq_list2.squeeze()
            I1_e = iq_list2[:,0]
            Q1_e = iq_list2[:,1]
            #print((Q1_e))                                                                                                                                                                      

            #crash4=crasher                                                                                                                                                                     


        #P_e = self.Correlation_P_e(I1_g, Q1_g, I2_g, Q2_g, I1_e, Q1_e)                                                                                                                         
        #fid_gg2, angle_gg2 = self.plot_results(I1_g, Q1_g, I2_g, Q2_g, self.QubitIndex)                                                                                                        
        fid_ge, angle_ge = self.plot_results2(I1_g, Q1_g, I2_g, Q2_g ,I1_e, Q1_e, self.QubitIndex)

        #sweep = Sweep("wait_loop", cfg['start'], cfg['stop'])                                                                                                                                  
        #print("---------------")                                                                                                                                                               
        #print(self.config)                                                                                                                                                                     
        #crash=crasher                                                                                                                                                                          
        correlation_t_calib = CorrelationGround_T_Swept(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay2'], cfg=self.config)
        iq_t_list = correlation_t_calib.acquire(self.experiment.soc, progress=True)
        #print(iq_t_list)                                                                                                                                                                       


        iq_t_rough = np.array(iq_t_list)
        iq_t_rough = iq_t_rough.squeeze()
        #print(iq_t_rough)                                                                                                                                                                      
        #iq_t = iq_t_rough.transpose(0,2,1)                                                                                                                                                     

        ground1, ground2 = iq_t_rough

        ground1_I = ground1[:,:,0]
        #print(ground1)                                                                                                                                                                         
        #print("----------")                                                                                                                                                                    
        ground1_Q = ground1[:,:,1]
        ground2_I = ground2[:,:,0]
        ground2_Q = ground2[:,:,1]

        #print(ground1_I[:,1])                                                                                                                                                                  

        #print(iq_t_rough[0,0])                                                                                                                                                                 

        iq_t_delay =correlation_t_calib.get_time_param('wait',"t", as_array=True)
        #print(iq_t_delay)                                                                                                                                                                      
        iq_t_delay = (np.array(iq_t_delay))- self.config["res_length"]
        #print(iq_t_delay)                                                                                                                                                                      

        g1_bar_t, g1_errors = self.Correlation_g1_bar_vs_time(ground1_I, ground1_Q, ground2_I, ground2_Q, I1_e, Q1_e, iq_t_delay)
        self.plot_results3(g1_bar_t,iq_t_delay,g1_errors ,self.QubitIndex)

        #P_e = self.Correlation_P_e(I1_g, Q1_g, I2_g, Q2_g, I1_e, Q1_e)                                                                                                                         
        ##fid_gg2, angle_gg2 = self.plot_results(I1_g, Q1_g, I2_g, Q2_g, self.QubitIndex)                                                                                                       
        #fid_ge, angle_ge = self.plot_results2(I1_g, Q1_g, I2_g, Q2_g ,I1_e, Q1_e, self.QubitIndex)                                                                                             
        P_e = 1/2 - ((1)/(2*np.sqrt(1 + 4 * g1_bar_t[0])))
        P_e_error_deriv = (1+ 4 * g1_bar_t[0])**(-1.5)
        P_e_error = g1_errors[0] * P_e_error_deriv


        return P_e, P_e_error , I1_g, Q1_g, I2_g , Q2_g, I1_e, Q1_e, self.config



    def plot_results(self, I1_g, Q1_g, I2_g, Q2_g, QubitIndex,  fig_quality=100):
        fid, threshold, angle, ig_new, ie_new = self.hist_ssf(I1_g, Q1_g, I2_g, Q2_g, cfg=self.config, plot=self.save_figs,  fig_quality=fig_quality)
        return fid, angle

    def plot_results2(self, I1_g, Q1_g, I2_g, Q2_g, I1_e, Q1_e, QubitIndex,  fig_quality=100):
        fid, threshold, angle, ig_new, ig2_new, ie_new = self.hist_ssf2(I1_g, Q1_g, I2_g, Q2_g, I1_e, Q1_e, cfg=self.config, plot=self.save_figs,  fig_quality=fig_quality)
        return fid, angle

    def plot_results3(self, g1_bar_t ,delay_times, g1_errors, QubitIndex, fig_quality=100):
        fig, axs = plt.subplots(figsize=(7, 4))
        fig.tight_layout()
        axs.errorbar(delay_times,g1_bar_t*100, yerr=g1_errors*100, label='g1_bar', color='b',capsize=4, marker='*', alpha=1.0, linestyle='None')
        axs.set_xlabel('delay time (us)')
        axs.set_ylabel('g_bar(t) (%)')
        axs.set_ylim(-0.1,2)
        axs.axhline(y=0,xmin=0,xmax=100, color='r', linestyle='--')
        axs.axvline(x=0, color='g', linestyle='--')
        self.create_folder_if_not_exists(self.outerFolder)
        outerFolder_expt = os.path.join(self.outerFolder, "q1_t_bar")
        self.create_folder_if_not_exists(outerFolder_expt)
        outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + self.expt_name+ f"{formatted_datetime}_"  + f"_q\{self.QubitIndex + 1}.png")
        axs.set_title(f"Correlator(%) vs delay time (us) " )
        fig.savefig(file_name,  dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)
        return




    def hist_ssf2(self, I1_g, Q1_g, I2_g, Q2_g, I1_e , Q1_e, cfg=None, plot=True,  fig_quality = 100):
        #print(I1_g)                                                                                                                                                                            

        ig = I1_g
        qg = Q1_g
        ig2 = I2_g
        qg2 = Q2_g
        ie = I1_e
        qe = Q1_e
        numbins = round(np.sqrt(float(cfg["reps2"])))

        xg, yg = np.median(ig), np.median(qg)
        xg2, yg2 = np.median(ig2), np.median(qg2)
        xe, ye = np.median(ie), np.median(qe)


        if plot == True:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
            fig.tight_layout()

            axs[0].scatter(ig, qg, label='g1', color='b', marker='*', alpha=0.2)
            axs[0].scatter(ig2, qg2, label='g2', color='g', marker='*', alpha=0.2)
            axs[0].scatter(ie, qe, label='e', color='r', marker = '*', alpha=0.2)
            axs[0].scatter(xg, yg, color='k', marker='o')
            axs[0].scatter(xg2, yg2, color='k', marker='o')
            axs[0].scatter(xe , ye, color= 'k', marker='o')

            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].legend(loc='upper right')
            axs[0].set_title(f'Unrotated: res_gain={self.config["res_gain_ge"]}\n res_len={self.config["res_length"]} us')
            axs[0].axis('equal')
        """Compute the rotation angle"""
        theta = -np.arctan2((ye - yg), (xe - xg))
        """Rotate the IQ data"""
        ig_new = ig * np.cos(theta) - qg * np.sin(theta)
        qg_new = ig * np.sin(theta) + qg * np.cos(theta)
        ig2_new = ig2 * np.cos(theta) - qg2 * np.sin(theta)
        qg2_new = ig2 * np.sin(theta) + qg2 * np.cos(theta)
        ie_new = ie * np.cos(theta) - qe * np.sin(theta)
        qe_new = ie * np.sin(theta) + qe * np.cos(theta)



        """New means of each blob"""
        xg, yg = np.median(ig_new), np.median(qg_new)
        xg2, yg2 = np.median(ig2_new), np.median(qg2_new)
        xe, ye = np.median(ie_new), np.median(qe_new)

        xlims = [np.min(ig_new), np.max(ie_new)]
        if plot == True:
            axs[1].scatter(ig_new, qg_new, label='g1', color='b', marker='*', alpha=0.2)
            axs[1].scatter(ig2_new, qg2_new, label='g2', color='g', marker='*', alpha=0.2)
            axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='*', alpha=0.2)
            axs[1].scatter(xg, yg, color='k', marker='o')
            axs[1].scatter(xg2, yg2, color='k', marker='o')
            axs[1].scatter(xe, ye, color='k', marker='o')
            axs[1].set_xlabel('I (a.u.)')
            axs[1].legend(loc='lower right')
            axs[1].set_title(f'Rotated Theta:{round(theta, 5)}:\n  res_gain={self.config["res_gain_ge"]}\n res_len={self.config["res_length"]} us')
            axs[1].axis('equal')
            """X and Y ranges for histogram"""
            ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g1', alpha=0.2)
            ng2, binsg2, pg2 = axs[2].hist(ig2_new, bins=numbins, range=xlims, color='g', label='g2', alpha=0.2)
            ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.2)
            axs[2].set_xlabel('I(a.u.)')
        else:
            ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
            ng2, binsg2 = np.histogram(ig2_new, bins=numbins, range=xlims)
            ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)




        """Compute the fidelity using overlap of the histograms"""
        contrast0 = np.abs(((np.cumsum(ng) - np.cumsum(ng2)) / (0.5 * ng.sum() + 0.5 * ng2.sum())))
        tind0 = contrast0.argmax()
        threshold0 = binsg[tind0]
        fid0 = contrast0[tind0]
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        contrast2 = np.abs(((np.cumsum(ng2) - np.cumsum(ne)) / (0.5 * ng2.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]
        tind2 = contrast2.argmax()
        threshold2 = binsg[tind2]
        fid2 = contrast2[tind2]


        if plot == True:
            self.create_folder_if_not_exists(self.outerFolder)
            outerFolder_expt = os.path.join(self.outerFolder, "IQ_g1_g2_e")
            self.create_folder_if_not_exists(outerFolder_expt)
            outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + self.expt_name+ f"{formatted_datetime}_"  + f"_q\{self.QubitIndex + 1}.png")
            axs[2].set_title(f"Fidelity (g1 : g2) = {fid0 * 100:.2f}% \n Fidelity (g1 : e) = {fid * 100:.2f}% \n Fidelity (ge : e) = {fid2 * 100:.2f}% " )
            fig.savefig(file_name,  dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
        return fid, threshold, theta, ig_new, ig2_new, ie_new


    def hist_ssf(self, I1_g, Q1_g, I2_g, Q2_g, cfg=None, plot=True,  fig_quality = 100):
        #print(I1_g)                                                                                                                                                                            

        ig = I1_g
        qg = Q1_g
        ig2 = I2_g
        qg2 = Q2_g
        numbins = round(np.sqrt(float(cfg["steps"])))

        xg, yg = np.median(ig), np.median(qg)
        xg2, yg2 = np.median(ig2), np.median(qg2)

        if plot == True:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
            fig.tight_layout()


            axs[0].scatter(ig, qg, label='g', color='b', marker='*', alpha=0.2)
            axs[0].scatter(ig2, qg2, label='e', color='r', marker='*', alpha=0.2)
            axs[0].scatter(xg, yg, color='k', marker='o')
            axs[0].scatter(xg2, yg2, color='k', marker='o')
            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].legend(loc='upper right')
            axs[0].set_title(f'Unrotated: res_gain={self.config["res_gain_ge"]}\n res_len={self.config["res_length"]} us')
            axs[0].axis('equal')
        """Compute the rotation angle"""
        theta = -np.arctan2((yg2 - yg), (xg2 - xg))
        """Rotate the IQ data"""
        ig_new = ig * np.cos(theta) - qg * np.sin(theta)
        qg_new = ig * np.sin(theta) + qg * np.cos(theta)
        ig2_new = ig2 * np.cos(theta) - qg2 * np.sin(theta)
        qg2_new = ig2 * np.sin(theta) + qg2 * np.cos(theta)

        """New means of each blob"""
        xg, yg = np.median(ig_new), np.median(qg_new)
        xg2, yg2 = np.median(ig2_new), np.median(qg2_new)

        # print(xg, xe)                                                                                                                                                                         
        xlims = [np.min(ig_new), np.max(ig2_new)]
        if plot == True:
            axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='*', alpha=0.2)
            axs[1].scatter(ig2_new, qg2_new, label='e', color='r', marker='*', alpha=0.2)
            axs[1].scatter(xg, yg, color='k', marker='o')
            axs[1].scatter(xg2, yg2, color='k', marker='o')
            axs[1].set_xlabel('I (a.u.)')
            axs[1].legend(loc='lower right')
            axs[1].set_title(f'Rotated Theta:{round(theta, 5)}:\n  res_gain={self.config["res_gain_ge"]}\n res_len={self.config["res_length"]} us')
            axs[1].axis('equal')


            """X and Y ranges for histogram"""
            ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.2)
            ng2, binsg2, pg2 = axs[2].hist(ig2_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.2)
            axs[2].set_xlabel('I(a.u.)')
        else:
            ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
            ng2, binsg2 = np.histogram(ig2_new, bins=numbins, range=xlims)

        """Compute the fidelity using overlap of the histograms"""
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ng2)) / (0.5 * ng.sum() + 0.5 * ng2.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]



        if plot == True:
            self.create_folder_if_not_exists(self.outerFolder)
            outerFolder_expt = os.path.join(self.outerFolder, "Correlation_g1_g2")
            self.create_folder_if_not_exists(outerFolder_expt)
            outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt,
                                     f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + self.expt_name+ f"{formatted_datetime}_"  + f"_q\                                                   
{self.QubitIndex + 1}.png")
            axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
            fig.savefig(file_name,  dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
        return fid, threshold, theta, ig_new, ig2_new


    def Correlation_P_e(self, data0, data1, data2, data3, data4, data5):
        import math
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import datetime

        # Unpack IQ data                                                                                                                                                                        
        ig = data0
        qg = data1
        ig2 = data2
        qg2 = data3
        ie = data4
        qe = data5


   
        # Find Mean of ground and excited state. This will be used for the zero'th order correlation function                                                                                  \
                                                                                                                                                                                                
        i_g0_g, q_g0_g = np.mean(ig), np.mean(qg)
        i_g0_e, q_g0_e = np.mean(ie), np.mean(qe)

        ## We recast our data to make our ground state -> 0 and the excited -> 1, This normalized our data                                                                                      

        delta_i_g0 = i_g0_e - i_g0_g
        delta_q_g0 = q_g0_e - q_g0_g

        v_bar_1 = np.real(((ig + (1j)*qg) - (i_g0_g + (1j)*q_g0_g))/(delta_i_g0 + (1j)*delta_q_g0))
        v_bar_2 = np.real(((ig2 + (1j)*qg2) - (i_g0_g + (1j)*q_g0_g))/(delta_i_g0 + (1j)*delta_q_g0))

        ## Now we calulxated the 1st order correlation function from this transformed data                                                                                                      

        g1_bar = np.mean(v_bar_1 * v_bar_2)

        ## From here we can then directly calculate P_e (see paper for derivation:https://arxiv.org/pdf/2001.00323)                                                                             

        P_e  = 0.5 * (1 - 1/(np.sqrt(1 + 4*g1_bar)))

        return P_e




    def Correlation_g1_bar_vs_time(self, data0, data1, data2, data3, data4, data5, delay_times):
        import math
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import datetime


        # Unpack excited IQ data                                                                                                                                                                
        ie = data4
        qe = data5
        i_g0_e, q_g0_e = np.mean(ie), np.mean(qe)

        #print(data0)                                                                                                                                                                           
        #print("-------------")                                                                                                                                                                 
        #print(data0[:,1])                                                                                                                                                                      
        #print("-------------")                                                                                                                                                                 
        #print(data0[:,2])                                                                                                                                                                      
        #crasher=crash                                                                                                                                                                          



        num_shots = len(data0[:,0])
        g1_bar_t = np.zeros(len(delay_times))
        g1_bar_t_error = np.zeros(len(delay_times))
        #unpack ground state and loop over delay times                                                                                                                                          
        for i in range(len(delay_times)):
            ig = data0[:,i]
            #crasheranalysis = crasher2                                                                                                                                                         
            qg = data1[:,i]
            ig2 = data2[:,i]
            qg2 = data3[:,i]
            i_g0_g, q_g0_g = np.mean(ig), np.mean(qg)
            ## We recast our data to make our ground state -> 0 and the excited -> 1, This normalized our data                                                                                  
            delta_i_g0 = i_g0_e - i_g0_g
            delta_q_g0 = q_g0_e - q_g0_g

            v_bar_1 = np.real(((ig + (1j)*qg) - (i_g0_g + (1j)*q_g0_g))/(delta_i_g0 + (1j)*delta_q_g0))
            v_bar_2 = np.real(((ig2 + (1j)*qg2) - (i_g0_g + (1j)*q_g0_g))/(delta_i_g0 + (1j)*delta_q_g0))

            ## Now we calulxated the 1st order correlation function from this transformed data                                                                                                  

            g1_bar_t[i] = np.mean(v_bar_1 * v_bar_2)
            print(num_shots)
            g1_bar_t_error[i] = np.std((v_bar_1 * v_bar_2), ddof=1) / np.sqrt(num_shots)

        return g1_bar_t, g1_bar_t_error


    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)




class CorrelationGround(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])


        self.add_gauss(ch=qubit_ch, name="ge_ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="pi_ge",style="arb",
                       envelope="ge_ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp']
                       )

        self.add_loop("shotloop", cfg["reps2"])
        #self.add_loop("shotloop", cfg["reps"])                                                                                                                                                 

    def _body(self, cfg): #this gives A_e                                                                                                                                                       
        #self.delay_auto(t=0.01, tag='waiting after pi')  # Wait til ge pi pulse is done before proceeding                                                                                      
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # probe pulse                                                                                                                      
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
        self.delay_auto(t=cfg['res_length']+0.1, tag = 'waiting for first readout to finish')
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # probe pulse                                                                                                                      
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])



class CorrelationExcited(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])


        self.add_gauss(ch=qubit_ch, name="ge_ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="pi_ge",
                       style="arb",
                       envelope="ge_ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_loop("shotloop", cfg["reps2"])
        #self.add_loop("shotloop", cfg["reps"])                                                                                                                                                 

    def _body(self, cfg): #this gives A_e                                                                                                                                                       
        self.pulse(ch=self.cfg["qubit_ch"], name="pi_ge", t=0)  # ge pulse                                                                                                                      
        self.delay_auto(t=0.0, tag='waiting')  # wait                                                                                                                                           
        self.delay_auto(t=0.01, tag='waiting after pi')  # Wait til ge pi pulse is done before proceeding                                                                                       
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # probe pulse                                                                                                                      
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
        #self.delay_auto(t=cfg['res_length']+0.01, tag = 'waiting for first readout to finish')                                                                                                 
        #self.pulse(ch=self.cfg["qubit_ch"], name="pi_ge", t=0)  # ge pulse                                                                                                                     
        #self.delay_auto(t=0.0, tag='waiting')  # wait                                                                                                                                          
        #self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # probe pulse                                                                                                                     
        #self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])                                                                                                                         





class CorrelationGround_T_Swept(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])


        self.add_gauss(ch=qubit_ch, name="ge_ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="pi_ge",
                       style="arb",
                       envelope="ge_ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )
        self.add_loop("shotloop", cfg["reps2"])


        #self.add_loop("waitloop", cfg["steps"])                                                                                                                                                
        #self.add_sweep(Sweep(name="waitsweep", parameter="wait_time", start=cfg['start'], stop=cfg['stop'], number=cfg['steps']))                                                              
        #print('t1 config in the program',cfg)                                                                                                                                                  
        self.add_loop("waitloop", cfg["steps"])
        #self.setup_step("waitloop", tag="wait", t=cfg['step_size'])                                                                                                                            
        #self.add_step("waitloop", "wait", t=cfg['step_size'])                                                                                                                                  
        #self.steps[0].add_step("wait", "t", cfg['step_size'])                                                                                                                                  


    def _body(self, cfg): #this gives A_e                                  )                                                                                                                    
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # Initial Measurement                                                                                                              
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
        #self.delay_auto(cfg['wait_time']+0.01, tag = 'wait')                                                                                                                                   
        self.delay_auto(cfg['wait_time']+1.3, tag='wait')
	#print(cfg['wait_time'])                                                                                                                                                                
        #print(cfg['wait_time'])                                                                                                                                                                
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # Correlation Second Measurement                                                                                                   
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
        #self.delay(3.0)                                                                                                                                                                        



