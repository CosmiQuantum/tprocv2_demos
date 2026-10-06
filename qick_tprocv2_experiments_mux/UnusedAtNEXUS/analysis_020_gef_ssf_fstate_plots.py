import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from analysis_001_plot_all_RR_h5 import PlotAllRR
import os
import sys
import h5py
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class GEF_SSF_ANALYSIS():
    def __init__(self, outerFolder_fstate, qubit_index, Analysis = True, RR = False, date_analysis = None, round_num = None):
        self.outerFolder_fstate = outerFolder_fstate
        self.QubitIndex = qubit_index
        self.date = date_analysis
        self.Analysis = Analysis
        self.RR = RR
        self.round_num = round_num

        if self.Analysis == True and self.RR == False: #To use this code for analysis instead of during RR
            if self.date is None:
                raise ValueError("For Analysis mode, date_analysis must be provided. What date does the data correspond to?")
            # Create a folder for saving the plots
            self.outerFolder_save_plots = os.path.join(self.outerFolder_fstate, self.date, "Study_Data", "PostProcessing_mode", "plots", "Q" + str(self.QubitIndex + 1)) # separated by qubit folder
            self.create_folder_if_not_exists(self.outerFolder_save_plots)
            # Create a folder for saving the data
            self.outerFolder_data = os.path.join(self.outerFolder_fstate, self.date, "Study_Data", "PostProcessing_mode", "Data_h5") #not separated by qubit folder
            self.create_folder_if_not_exists(self.outerFolder_data)

        elif self.Analysis == False and self.RR == True: #to use this code during RR
            if self.round_num is None: #used for saving the files at the end
                raise ValueError("For round robin mode, round_num must be provided (by the round robin script).")
            # Create a folder for saving the plots
            self.outerFolder_save_plots = os.path.join(self.outerFolder_fstate, str(datetime.date.today()), "Study_Data", "Round_Robin_mode", "plots", "Q" + str(self.QubitIndex + 1)) # separated by qubit folder
            self.create_folder_if_not_exists(self.outerFolder_save_plots)
            # Create a folder for saving the data
            self.outerFolder_data = os.path.join(self.outerFolder_fstate, str(datetime.date.today()),"Study_Data", "Round_Robin_mode", "Data_h5") #not separated by qubit folder
            self.create_folder_if_not_exists(self.outerFolder_data)

        else:
            raise ValueError("You must choose either Analysis mode or RR mode (one must be true and the other false).")


    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def parse_iq_string(self, dataset):
        """
        Convert a bracketed byte/string (e.g. b'[1.2 3.4 5.6]') into a float NumPy array.
        """
        # If it's bytes, decode to a string
        if isinstance(dataset, bytes):
            dataset = dataset.decode()
        # Remove leading/trailing whitespace
        dataset = dataset.strip()
        # Remove any leading/trailing brackets
        dataset = dataset.strip('[]')
        # Replace line breaks with spaces (if any)
        dataset = dataset.replace('\n', ' ')
        # Convert to a float array using space separation
        arr = np.fromstring(dataset, sep=' ')
        return arr

    def robust_center_and_sigma(self, points):
        """
        Given a 2D array 'points' (each row is [I, Q]), return the robust center (median)
        and a robust sigma computed from the radial distances using the median absolute deviation (MAD)
        scaled by 1.4826.
        """
        center = np.median(points, axis=0)
        # Compute radial distances from the center.
        distances = np.linalg.norm(points - center, axis=1)
        # Robust sigma: scale the median absolute deviation of the distances.
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
        robust_sigma = 1.4826 * mad
        return center, robust_sigma


    def f_outside_tangent_line(self, plot_data, sigma_num):
        """
        Uses the e-state data (I_e, Q_e) to compute a robust center and sigma circle (radius = sigma_num * robust sigma).
        Then, it computes the direction from the e-state center to the f-state center (using the median of f-state data).
        It defines the tangent line to the e-state circle at the point T = center_e + (radius_e * v), where v is the unit vector
        from e-state center to f-state center. The half-plane defined by the tangent line that does NOT contain the e-state
        circle is determined by the condition: dot((x - T), v) > 0.

        Returns:
          center_e (ndarray): Robust e-state center.
          radius_e (float): Computed radius (sigma_num * robust sigma).
          T (ndarray): Tangency point on the e-state circle.
          v (ndarray): Unit vector from center_e to f-state center.
          f_outside (ndarray): f-state points that lie in the half-plane outside the tangent line.
        """
        # Compute robust center and sigma for e-state data
        points_e = np.column_stack((plot_data['I_e'], plot_data['Q_e']))
        center_e, robust_sigma_e = self.robust_center_and_sigma(points_e)
        radius_e = sigma_num * robust_sigma_e

        # Compute robust center for f-state data
        points_f = np.column_stack((plot_data['I_f'], plot_data['Q_f']))
        center_f = np.median(points_f, axis=0)

        # Compute direction from e-state center to f-state center.
        v = center_f - center_e
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            v = np.array([1, 0])
        else:
            v = v / norm_v

        # Compute the tangency point on the e-state circle.
        T = center_e + radius_e * v

        # For each f-state point, compute dot((x - T), v).
        dots = np.dot(points_f - T, v)
        # Points for which this dot product is > 0 are on the side of the tangent line not containing the e-state circle.
        outside_mask = dots > 0
        f_outside = points_f[outside_mask]

        return center_e, radius_e, T, v, f_outside

    def fstate_analysis_plot(self, I_g, Q_g, I_e, Q_e, I_f, Q_f, ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge, qubit_index, provided_sigma_num = None):
        # ----------------------------------------------------------Process rotated and unrotated data-------------------------------------------------
        plot_data = {
            'I_g': I_g,
            'Q_g': Q_g,
            'I_e': I_e,
            'Q_e': Q_e,
            'I_f': I_f,
            'Q_f': Q_f}

        rotated_data = {
            'I_e': ie_new,
            'Q_e': qe_new,
            'I_g': ig_new,
            'Q_g': qg_new,
            'I_f': if_new,
            'Q_f': qf_new}

        # Build the rotated e-state points array from the original rotated data
        points_e_rot = np.column_stack((ie_new, qe_new))
        # Filter rotated e-state points after threshold_ge from the original data
        filtered_epoints = points_e_rot[points_e_rot[:, 0] >= threshold_ge]
        # Compute the median center of the filtered points:
        center_filtered = np.median(filtered_epoints, axis=0)

        # Filter rotated f-state points beyond the threshold and past the e-state circle center (center_filtered)
        # --- Build the rotated f-state points ---
        points_f_rot = np.column_stack((rotated_data['I_f'], rotated_data['Q_f']))
        # First filter by threshold:
        mask_threshold = points_f_rot[:, 0] >= threshold_ge
        filtered_f_rot = points_f_rot[mask_threshold]
        # --- Now filter them based on angle ---
        # Compute the vectors from the e-state center to each filtered f-state point:
        rel_vectors = filtered_f_rot - center_filtered
        # Compute the angle (in radians) for each relative vector. arctan2 returns values in (-pi, pi].
        angles = np.arctan2(rel_vectors[:, 1], rel_vectors[:, 0])
        # Define an angular cutoff
        cutoff = np.deg2rad(135) #np.pi / 2
        # Create a mask that keeps only points whose absolute angle is less than the cutoff.
        mask_angle = np.abs(angles) < cutoff

        #secondary height mask for those points >cutoff : only include those whose y is well above the g-state median
        points_g_rot = np.column_stack((ig_new, qg_new))
        g_median_y = np.median(points_g_rot[:, 1])
        height_margin = 2.0  # adjust as needed
        mask_height = (np.abs(angles) >= cutoff) & (filtered_f_rot[:, 1] >= g_median_y + height_margin)

        #combine masks
        mask_combined = mask_angle | mask_height #this does mask_angle[i] OR mask_height[i]

        #final filtered f-state points
        filtered_f_rot_final = filtered_f_rot[mask_combined]

        filtered_I_f_rot = filtered_f_rot_final[:, 0]
        filtered_Q_f_rot = filtered_f_rot_final[:, 1]
        filtered_f_rot = np.column_stack((filtered_I_f_rot, filtered_Q_f_rot))

        # Inverse rotation by -theta_ge, for unrotated data
        cos_t = np.cos(theta_ge)
        sin_t = np.sin(theta_ge)
        # For each (I',Q') in filtered_f_rot, compute:
        filtered_f_unrot = np.empty_like(filtered_f_rot)
        filtered_f_unrot[:, 0] = filtered_f_rot[:, 0] * cos_t + filtered_f_rot[:, 1] * sin_t
        filtered_f_unrot[:, 1] = -filtered_f_rot[:, 0] * sin_t + filtered_f_rot[:, 1] * cos_t

        # Filtered unrotated data: use only those inverse-rotated f-points
        filtered_plot_data = {
            'I_g': plot_data['I_g'],
            'Q_g': plot_data['Q_g'],
            'I_e': plot_data['I_e'],
            'Q_e': plot_data['Q_e'],
            'I_f': filtered_f_unrot[:, 0],
            'Q_f': filtered_f_unrot[:, 1],
        }
        # Rotated data: use the filtered_rotated_data as before
        filtered_rotated_data = {
            'I_e': rotated_data['I_e'],
            'Q_e': rotated_data['Q_e'],
            'I_g': rotated_data['I_g'],
            'Q_g': rotated_data['Q_g'],
            'I_f': filtered_I_f_rot,
            'Q_f': filtered_Q_f_rot,
        }

        # Assign sigma_num (if provided it will use the provided value, if not provided it will calculate an appropriate one)
        # Reminder: sigma_num determines the size of the e-state circle radius, that's it.
        if provided_sigma_num is None: #user did not provide a sigma_num, wants dynamic sigma approach
            # Compute the median center of the filtered points:
            center_filtered = np.median(filtered_epoints, axis=0)
            # Compute distances of filtered points from that median:
            distances = np.linalg.norm(filtered_epoints - center_filtered, axis=1)
            # chosen percentile of these distances:
            perc = np.percentile(distances, 70) #larger than 80 can cause problems, direction of f-state gets confused
            # if self.QubitIndex == 3: #qubit 4 is more sensitive
            #     perc = np.percentile(distances,70)  # larger than 80 can cause problems, direction of f-state gets confused
            # Compute robust sigma for the filtered points:
            median_dist = np.median(distances)
            mad = np.median(np.abs(distances - median_dist))
            robust_sigma_filtered = 1.4826 * mad
            # Determine dynamic sigma multiplier:
            sigma_num = perc / robust_sigma_filtered if robust_sigma_filtered != 0 else 1
        elif provided_sigma_num is not None: #user provided a sigma_num value
            sigma_num = provided_sigma_num  # use provided value

        #--------------calculate rotated data circle parameters using assigned sigma_num------------------------------------------
        center_e_rot, radius_e_rot, T_rot, v_rot, f_outside_rot = self.f_outside_tangent_line(filtered_rotated_data, sigma_num)


        # -------------calculate unrotated data circle parameters using final assigned sigma_num------
        center_e, radius_e, T, v, f_outside = self.f_outside_tangent_line(filtered_plot_data, sigma_num)
        line_length = radius_e * 3
        tangent_direction = np.array([-v[1], v[0]])  # Perpendicular to v
        line_point1 = T + tangent_direction * line_length
        line_point2 = T - tangent_direction * line_length

        #------------------Check if the rotated e-state circle crosses the g-e threshold that is used to calculate g-e fidelities---------------
        if (center_e_rot[0] - radius_e_rot) < threshold_ge:

            # Shift the center's x-coordinate so that left edge equals threshold_ge and a little bit more if needed
            center_e_rot[0] = threshold_ge + radius_e_rot + 0.5

            #Recalculate v_rot
            points_f_rot = np.column_stack((
                filtered_rotated_data['I_f'],
                filtered_rotated_data['Q_f']
            ))
            center_f_rot_new = np.median(points_f_rot, axis=0)
            v_rot = center_f_rot_new - center_e_rot
            norm_v_rot = np.linalg.norm(v_rot)
            if norm_v_rot != 0:
                v_rot = v_rot / norm_v_rot
            else:
                v_rot = np.array([1, 0])

            # recalc the tangency point T_rot:
            T_rot = center_e_rot + radius_e_rot * v_rot # v_rot remains the same

            # recalc f_outside for the rotated data
            dots_rot = np.dot(points_f_rot - T_rot, v_rot)
            outside_mask_rot = dots_rot > 0
            f_outside_rot = points_f_rot[outside_mask_rot]

            # adjust the unrotated circle and tangent line too
            center_e = np.array([
                center_e_rot[0] * np.cos(-theta_ge) - center_e_rot[1] * np.sin(-theta_ge),
                center_e_rot[0] * np.sin(-theta_ge) + center_e_rot[1] * np.cos(-theta_ge)])

            # Recalculate v for the unrotated data
            points_f = np.column_stack((
                filtered_plot_data['I_f'],
                filtered_plot_data['Q_f']
            ))
            center_f_new = np.median(points_f, axis=0)
            v = center_f_new - center_e
            norm_v = np.linalg.norm(v)
            if norm_v != 0:
                v = v / norm_v
            else:
                v = np.array([1, 0])

            T = center_e + radius_e * v  # v remains the same

            # recalc line endpoints for unrotated data
            line_length = radius_e * 3
            tangent_direction = np.array([-v[1], v[0]])
            line_point1 = T + tangent_direction * line_length
            line_point2 = T - tangent_direction * line_length

            # update f_outside for unrotated data
            dots = np.dot(points_f - T, v)
            outside_mask = dots > 0
            f_outside = points_f[outside_mask]


        #Continue with the rotated data calculations
        line_length_rot = radius_e_rot * 3
        tangent_direction_rot = np.array([-v_rot[1], v_rot[0]])
        line_point1_rot = T_rot + tangent_direction_rot * line_length_rot
        line_point2_rot = T_rot - tangent_direction_rot * line_length_rot

        # print("~ e-state center:", center_e)
        # print("Computed {}s radius: {:.2f}".format(sigma_num, radius_e))
        # print("Number of f-state points outside tangent line:", len(f_outside))
        # if len(f_outside) > 0:
        #     print("Median location of f-state points outside tangent line:", np.median(f_outside, axis=0))
        # print("Tangent line endpoints in IQ space:")
        # print("Endpoint 1:", line_point1)
        # print("Endpoint 2:", line_point2, "\n")

        # if len(f_outside_rot) > 0:
        #     print("Median location of rotated f-state points outside tangent line:", np.median(f_outside_rot, axis=0))
        #
        # # Print the endpoints
        # print("Tangent line endpoints in rotated IQ space:")
        # print("Endpoint 1:", line_point1_rot)
        # print("Endpoint 2:", line_point2_rot)

        #-----------Get time stamp for when this f-state location in IQ space was determined ----------------------
        time_stamp = time.mktime(datetime.datetime.now().timetuple())

        # ---------------------Plot two subplots, of equal width, showing the rotated and unrotated data with the identified f-state data-----------------------
        # Define number of bins for histogram (you can adjust as needed)
        numbins = 45

        # Compute x-range for the histogram using rotated data (you can adjust this if needed)
        xlims = [np.min(if_new), np.max(if_new)]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1, 1]})
        fig.suptitle(f"Qubit {self.QubitIndex + 1} f-state Points Outside e-state {sigma_num:.2f}-sigma Circle Tangent", fontsize=16)

        # ---- Unrotated Data Plot (Left) ----
        points_g = np.column_stack((plot_data['I_g'], plot_data['Q_g']))
        ax1.scatter(points_g[:, 0], points_g[:, 1], color='blue', label='g-state data', alpha=0.8)
        points_e = np.column_stack((plot_data['I_e'], plot_data['Q_e']))
        ax1.scatter(points_e[:, 0], points_e[:, 1], color='red', label='e-state data', alpha=0.6)
        points_f = np.column_stack((plot_data['I_f'], plot_data['Q_f']))
        ax1.scatter(points_f[:, 0], points_f[:, 1], color='green', label='f-state data', alpha=0.4)
        if len(f_outside) > 0:
            ax1.scatter(f_outside[:, 0], f_outside[:, 1], color='black', marker='.', s=100,
                        label='f-state outside tangent')
        circle_e = plt.Circle(center_e, radius_e, color='black', fill=False, linestyle='--',
                              label='e-state {:.2f}-sigma circle'.format(sigma_num))
        ax1.add_patch(circle_e)
        ax1.plot([line_point1[0], line_point2[0]], [line_point1[1], line_point2[1]], color='black', linestyle='-',
                 label='Tangent line')
        ax1.set_xlabel("I (a.u.)")
        ax1.set_ylabel("Q (a.u.)")
        ax1.set_title("Unrotated")
        ax1.axis('equal')

        # ---- Rotated Data Plot (middle) ----
        points_g_rot = np.column_stack((ig_new, qg_new))
        ax2.scatter(points_g_rot[:, 0], points_g_rot[:, 1], color='blue', label='Rotated g-state data', alpha=0.6)
        points_e_rot = np.column_stack((ie_new, qe_new))
        ax2.scatter(points_e_rot[:, 0], points_e_rot[:, 1], color='red', label='Rotated e-state data', alpha=0.6)
        points_f_rot = np.column_stack((if_new, qf_new))
        ax2.scatter(points_f_rot[:, 0], points_f_rot[:, 1], color='green', label='Rotated f-state data', alpha=0.6)
        if len(f_outside_rot) > 0:
            ax2.scatter(f_outside_rot[:, 0], f_outside_rot[:, 1], color='black', marker='.', s=100,
                        label='Rotated f-state outside tangent')
        circle_e_rot = plt.Circle(center_e_rot, radius_e_rot, color='black', fill=False, linestyle='--',
                                  label='Rotated e-state {}s circle'.format(sigma_num))
        ax2.add_patch(circle_e_rot)
        ax2.plot([line_point1_rot[0], line_point2_rot[0]], [line_point1_rot[1], line_point2_rot[1]], color='black',
                 linestyle='-', label='Rotated Tangent line')
        ax2.set_xlabel("I' (a.u.)")
        ax2.set_ylabel("Q' (a.u.)")
        ax2.set_title(f'Rotated (theta_ge: {round(theta_ge, 5)})')
        ax2.axis('equal')

        # ---- Histogram Plot (Right) ----
        # Plot histograms of the rotated I-values for g, e, and f data.
        # Here we use the same x-range (xlims) for all histograms.
        ng, binsg, _ = ax3.hist(ig_new, bins=numbins, range=xlims, color='blue', label='g', alpha=0.8)
        ne, binse, _ = ax3.hist(ie_new, bins=numbins, range=xlims, color='red', label='e', alpha=0.6)
        nf, binsf, _ = ax3.hist(if_new, bins=numbins, range=xlims, color='green', label='f', alpha=0.4)
        ax3.set_xlabel("I' (a.u.)")
        ax3.set_title("Histogram of I'")

        # ---add a common legend for the plots----
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles = handles1
        labels = labels1

        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.86, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        #-----------------Save the figure as a PNG file-----------------------------------
        formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self.Analysis == True and self.RR == False: #analysis mode
            png_filename = os.path.join(self.outerFolder_save_plots, f"Q{qubit_index+1}_SSF_fstate_IQspace_{formatted_datetime}.png")
            fig.savefig(png_filename, dpi=200, bbox_inches='tight')
            # print(f"Figure saved to: {self.outerFolder_save_plots}")


            h5_filename = os.path.join(self.outerFolder_data, f"Q{qubit_index+1}_SSF_fstate_IQspace_{formatted_datetime}.h5")
            # print(f"Data saved to: {self.outerFolder_data}")

            mode_str = "post_processing_mode"

        elif self.Analysis == False and self.RR == True:  # RR mode
            png_filename = os.path.join(self.outerFolder_save_plots, f"Q{qubit_index + 1}_SSF_fstate_IQspace_{formatted_datetime}_Round_{self.round_num}.png")
            fig.savefig(png_filename, dpi=200, bbox_inches='tight')
            # print(f"Figure saved to: {self.outerFolder_save_plots}")

            h5_filename = os.path.join(self.outerFolder_data, f"Q{qubit_index + 1}_SSF_fstate_IQspace_{formatted_datetime}_Round_{self.round_num}.h5")
            # print(f"Data saved to: {self.outerFolder_data}")

            mode_str = "round_robin_mode"

        else:
            raise ValueError("You must choose either Analysis mode or RR mode (one must be true and the other false in the Class Initialization).")

        with h5py.File(h5_filename, 'w') as hf:
            hf.create_dataset('timestamp', data=time_stamp)
            # Save tangent info and line lengths
            hf.create_dataset('line_length', data=line_length)
            hf.create_dataset('tangent_direction', data=tangent_direction)
            hf.create_dataset('line_length_rot', data=line_length_rot)
            hf.create_dataset('tangent_direction_rot', data=tangent_direction_rot)

            # Save unrotated calculated parameters
            hf.create_dataset('line_point1', data=line_point1)
            hf.create_dataset('line_point2', data=line_point2)
            hf.create_dataset('center_e', data=center_e)
            hf.create_dataset('radius_e', data=radius_e)
            hf.create_dataset('T', data=T)
            hf.create_dataset('v', data=v)
            hf.create_dataset('f_outside', data=f_outside)

            # Save rotated calculated parameters
            hf.create_dataset('line_point1_rot', data=line_point1_rot)
            hf.create_dataset('line_point2_rot', data=line_point2_rot)
            hf.create_dataset('center_e_rot', data=center_e_rot)
            hf.create_dataset('radius_e_rot', data=radius_e_rot)
            hf.create_dataset('T_rot', data=T_rot)
            hf.create_dataset('v_rot', data=v_rot)
            hf.create_dataset('f_outside_rot', data=f_outside_rot)

            # Save input arrays and parameters
            hf.create_dataset('I_g', data=I_g)
            hf.create_dataset('Q_g', data=Q_g)
            hf.create_dataset('I_e', data=I_e)
            hf.create_dataset('Q_e', data=Q_e)
            hf.create_dataset('I_f', data=I_f)
            hf.create_dataset('Q_f', data=Q_f)
            hf.create_dataset('ig_new', data=ig_new)
            hf.create_dataset('qg_new', data=qg_new)
            hf.create_dataset('ie_new', data=ie_new)
            hf.create_dataset('qe_new', data=qe_new)
            hf.create_dataset('if_new', data=if_new)
            hf.create_dataset('qf_new', data=qf_new)
            hf.create_dataset('theta_ge', data=theta_ge)
            hf.create_dataset('threshold_ge', data=threshold_ge)
            hf.create_dataset('sigma_num', data=sigma_num)

            # Save the qubit index
            hf.create_dataset('qubit_index', data=qubit_index)

            # Save the mode
            hf.create_dataset('mode', data=mode_str)

            # If round-robin mode, save round number as well
            if self.RR:
                hf.create_dataset('round_num', data=self.round_num)


        return line_point1, line_point2, center_e, radius_e, T, v, f_outside, line_point1_rot, line_point2_rot, center_e_rot, radius_e_rot, T_rot, v_rot, f_outside_rot
