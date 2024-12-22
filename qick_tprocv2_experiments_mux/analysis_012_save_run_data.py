import json
import h5py
import os

class SaveRunData:
    def __init__(self, run_number, run_notes):
        self.run_number = run_number
        self.run_notes = run_notes

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def run(self,gaussian_dates, t1_vals, t1_errs, t1_std_values, t1_mean_values, t2r_vals, t2r_errs,
                 t2r_mean_values, t2r_std_values, t2e_vals, t2e_errs, t2e_mean_values, t2e_std_values):
        ######## save run statistics to the run_stats folder so we can compare/plot to previous runs later #############

        # t1_vals and t1_errs are lists each containing N lists with data, where N is number of qubits
        # std_values and mean_values are dictionaries each with the mean/std values for each qubit
        # run_number is an int, it starts from the first run with qubits
        # run_notes is a string containing brief information about updates to the setup this run
        # last_date is the last date and time saved, use this to say what date the run was on

        last_date = gaussian_dates[max(gaussian_dates.keys())][-1] if gaussian_dates[
            max(gaussian_dates.keys())] else None

        run_stats_folder = f"run_stats/run{self.run_number}/"
        self.create_folder_if_not_exists(run_stats_folder)
        filename = run_stats_folder + 'experiment_data.h5'

        with h5py.File(filename, 'w') as hf:
            # Save dictionaries as JSON strings in attributes
            hf.attrs['t1_vals'] = json.dumps(t1_vals)
            hf.attrs['t1_errs'] = json.dumps(t1_errs)

            hf.attrs['t1_std_values'] = json.dumps(t1_std_values)
            hf.attrs['t1_mean_values'] = json.dumps(t1_mean_values)

            # Save run_number as an attribute
            hf.attrs['run_number'] = self.run_number

            # Save run_notes and last_date as attributes
            hf.attrs['run_notes'] = self.run_notes
            hf.attrs['last_date'] = last_date

            hf.attrs['t2r_vals'] = json.dumps(t2r_vals)
            hf.attrs['t2r_errs'] = json.dumps(t2r_errs)

            hf.attrs['t2r_mean_values'] = json.dumps(t2r_mean_values)
            hf.attrs['t2r_std_values'] = json.dumps(t2r_std_values)

            hf.attrs['t2e_vals'] = json.dumps(t2e_vals)
            hf.attrs['t2e_errs'] = json.dumps(t2e_errs)
        
            hf.attrs['t2e_mean_values'] = json.dumps(t2e_mean_values)
            hf.attrs['t2e_std_values'] = json.dumps(t2e_std_values)