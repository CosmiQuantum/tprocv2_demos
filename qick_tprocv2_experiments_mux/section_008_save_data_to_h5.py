import datetime
import numpy as np
import h5py
import os
np.set_printoptions(threshold=1000000000000000)
import numbers
import json
import datetime as date

class Data_H5:
    def __init__(self, outerFolder, data = None, batch_num = 0, save_r = 0):
        self.outerFolder_expt = outerFolder
        self.data = data
        self.batch_num = batch_num
        self.save_r = save_r

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def convert_non_floats_to_strings(self, data_list):
        return [str(x) if not isinstance(x, (int, float, np.float64)) else x for x in data_list]

    def create_dataset(self, name, value, group):
        if value is not None:
            if name == "Dates":  # Handle timestamps directly
                try:
                    value = np.array(value, dtype=np.float64)  # Store as floats
                except ValueError as e:
                    #print(f"Warning: Could not convert dates for '{name}'. Error: {e}")
                    value = np.array(value, dtype='S')  # Fallback to string if needed
            elif isinstance(value, list) and 'None' in str(value[0]):
                value = np.array([])
            else:
                try:
                    # need to do this because sometimes a list has not all floats, but a float and sometimes a list
                    value = self.convert_non_floats_to_strings(value)
                    value = np.array(value, dtype=np.float64)
                except ValueError:
                    #print(f"Warning: Could not convert data for '{name}' to float. Saving as string.")
                    value = np.array(value, dtype='S')
            group.create_dataset(name, data=value)
        else:
            group.create_dataset(name, data=np.array([]))

    def create_dataset_clean(self, name, value, group):
        ## if the value is None just save it like that
        if value is None:
            group.create_dataset(name, data=np.array([]))
            return

        ## if its an array save it like an array, otherwise its a ragged array
        if isinstance(value, np.ndarray):
            ##safe to store
            if value.dtype != object:
                group.create_dataset(name, data=value)
                return

            ## ragged list
            self.create_dataset_clean(name, value.tolist(), group)
            return

        ## if its a boolean save it like an attribute
        if isinstance(value, (numbers.Number, bool, np.bool_)):
            group.attrs[name] = value
            return

        ## if its a datetime or date save with utf-8 encoding for string
        if isinstance(value, (date.datetime, date.date, np.datetime64)):
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(name, data=np.array(value.isoformat(), dtype=dt))
            return

        ## if a string save it like one
        if isinstance(value, str):
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(name, data=np.array(value, dtype=dt))
            return

        ## now if its a list lets try a few things in case it's a ragged list
        if isinstance(value, (list, tuple)):
            ## see if it's all numbers inside of the list
            try:
                arr = np.asarray(value, dtype=float)  ## check if you cna convert it to an array with al floats
                if arr.dtype != object:  ## if it fails it wont be a real array and this wont work
                    group.create_dataset(name, data=arr) ## if it succeeds it will be a real object and will get here and save
                    return
            except Exception:
                pass ## keep going to the other methods if this fails

            ## if it's a list of strings or has a None in there. if theres a None make it a string
            if all(isinstance(x, str) or x is None for x in value):
                dt = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(name,
                                     data=[x if x is not None else '' for x in value],
                                     dtype=dt)
                return
            else:
                ## if everything else failed then you have a very ragged list. Give up and just store as a json string
                dt = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(name, data=[json.dumps(elem, default=str) for elem in value], dtype=dt)
                return

        ## if its a dictionary, it could have ragged lists, so call this definition again on the values for proper handling
        if isinstance(value, dict):
            subgrp = group.create_group(name)
            for sub_name, sub_val in value.items():
                self.create_dataset(sub_name, sub_val, subgrp)
            return

        ## if none of those worked, do a json string again
        json_str = json.dumps(value, default=str)
        dt = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=np.array(json_str, dtype=dt))

    def save_to_h5(self, data_type, save_dataset_clean=False, additional_title = ''):
        self.outerFolder_expt = os.path.join(self.outerFolder_expt, "Data_h5", f"{data_type}")
        self.create_folder_if_not_exists(self.outerFolder_expt)
        formatted_datetime =  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        h5_filename = os.path.join(self.outerFolder_expt, f"{formatted_datetime}_" + f"{data_type}_results_batch_{self.batch_num}_" + f"Num_per_batch{self.save_r}{additional_title}.h5")
        with h5py.File(h5_filename, 'w') as f:
            f.attrs['datestr']=formatted_datetime
            f.attrs[f'{data_type}_results_batch']=self.batch_num
            f.attrs['num_per_batch'] = self.save_r
            for QubitIndex, data in self.data.items():
                # Create a group for each qubit
                group = f.create_group(f'Q{QubitIndex + 1}')

                #grab keys and get data
                for key, value in data.items():
                    if value is not None:  # check for None values before creating dataset
                        if save_dataset_clean:
                            self.create_dataset_clean(key, value, group)
                        else:
                            self.create_dataset(key, value, group)


        #self.print_h5_contents(h5_filename)
        #print(h5_filename)

    def print_h5_contents(self, filename):
        try:
            with h5py.File(filename, 'r') as f:
                print("File Attributes:")
                for key, value in f.attrs.items():
                    print(f"  {key}: {value}")
                print("\nDataset Contents:")
                for key in f.keys():
                    group = f[key]
                    print(f"\nGroup: {key}")
                    for dataset_name in group.keys():
                        dataset = group[dataset_name]
                        print(f"  Dataset: {dataset_name}")
                        try:  # Handle various datatypes gracefully
                            data = dataset[()]
                            if isinstance(data,
                                          np.ndarray) and data.size > 100:  # Check array size for printing only sample
                                print(f"    Data (sample): {data[:50]} ... (truncated)")
                            else:
                                print(f"    Data: {data}")
                        except:
                            print("    Data: Could not print data")  # Catch potential errors printing complex objects

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def load_from_h5(self, data_type,  save_r=1):  # Added save_r as parameter.
        """Loads data from an HDF5 file into specified dictionary format."""
        mode = 'r'
        if data_type == 'Qtemps': #to switch from old naming convention to new naming convention for qubit temp data
            data_type = 'q_temperatures'
            mode = 'r+'

        data = {data_type: {}}  # Initialize the main dictionary with the data_type.

        with h5py.File(self.outerFolder_expt, mode) as f:
            for qubit_group in f.keys():
                qubit_index = int(qubit_group[1:]) - 1
                qubit_data = {}
                group = f[qubit_group]

                if 'Qtemps' in group and 'q_temperatures' not in group: #to switch from old naming convention to new naming convention for qubit temp data
                    group.move('Qtemps', 'q_temperatures')

                for dataset_name in group.keys():
                    # Attempt to map HDF5 keys to the target dictionaries' keys.
                    if data_type == 'Res' or data_type == 'Res_ge' or  data_type == 'Res_ef':
                        target_keys = {'Dates': 'Dates', 'freq_pts': 'freq_pts', 'freq_center': 'freq_center',
                                       'Amps': 'Amps', 'Found Freqs': 'Found Freqs', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                       'Syst Config': 'Syst Config'}
                    elif data_type == 'QSpec' or data_type == 'QSpec_ge' or  data_type == 'QSpec_ef' or data_type == 'qspec_ge':
                        target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Frequencies': 'Frequencies',
                                       'I Fit': 'I Fit', 'Q Fit': 'Q Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num', 'Recycled QFreq': 'Recycled QFreq',
                                       'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                    elif data_type == 'Rabi' or data_type == 'Rabi_ge' or  data_type == 'Rabi_ef':
                        target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Gains': 'Gains', 'Fit': 'Fit',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                       'Syst Config': 'Syst Config'}

                    elif data_type == 'q_temperatures':
                        target_keys = {'Dates': 'Dates', 'Qfreq_ge': 'Qfreq_ge',
                                       'I1': 'I1', 'Q1': 'Q1', 'Gains1': 'Gains1', 'Fit1': 'Fit1',
                                       'I2': 'I2', 'Q2': 'Q2', 'Gains2': 'Gains2', 'Fit2': 'Fit2',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                       'Syst Config': 'Syst Config'}

                    elif data_type == 'Rabi_QZE':
                        target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Mag': 'Mag', 'Gains': 'Gains', 'Fit': 'Fit',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                       'Syst Config': 'Syst Config'}
                    elif data_type == 'SS':
                        target_keys = {'Fidelity': 'Fidelity', 'Angle': 'Angle', 'Dates': 'Dates', 'I_g': 'I_g',
                                       'Q_g': 'Q_g', 'I_e': 'I_e', 'Q_e': 'Q_e',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                       'Syst Config': 'Syst Config'}

                    elif data_type == 'SS_gef':
                        target_keys = {'Fidelity': 'Fidelity', 'Angle_ge': 'Angle_ge', 'Dates': 'Dates', 'I_g': 'I_g', 'Q_g': 'Q_g', 'I_e': 'I_e', 'Q_e': 'Q_e', 'I_f': 'I_f', 'Q_f': 'Q_f', 'Round Num': 'Round Num', 'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                        'Syst Config': 'Syst Config'}
                    elif data_type == 'T1' or data_type == 'T1_ge' or data_type == 't1_ge' or  data_type == 'T1_fe' or  data_type == 'T1_fg':
                        target_keys = {'T1': 'T1', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                       'Syst Config': 'Syst Config'}
                    elif data_type == 'T2':
                        target_keys = {'T2': 'T2', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                       'Syst Config': 'Syst Config'}
                    elif data_type == 'T2E':
                        target_keys = {'T2E': 'T2E', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config',
                                       'Syst Config': 'Syst Config'}
                    else:
                        raise ValueError(f"Unsupported data_type: {data_type}")

                    try:
                        mapped_key = target_keys[dataset_name]  # Map HDF5 key to target key.
                        qubit_data[mapped_key] = [group[dataset_name][()]] * save_r  # Expand to match the desired length.

                    except KeyError:
                        print(
                            f"Warning: Key '{dataset_name}' not found in target dictionary for data_type '{data_type}'. Skipping.")
                        pass

                data[data_type][qubit_index] = qubit_data

        return data

    def convert_for_hdf5(self, data):
        if isinstance(data, dict):
            return {k: self.convert_for_hdf5(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_for_hdf5(x) for x in data]
        elif isinstance(data, np.ndarray):
            if data.dtype == 'O':
                try:
                    # Attempt to convert to string type. If it fails, raise an error
                    return np.array(data, dtype='U')
                except:
                    raise TypeError(f"Cannot convert NumPy object array to HDF5-compatible type: {data}")
            else:
                return data  # Already a suitable type
        elif isinstance(data, object):  # Added to handle objects
            try:
                return str(data)  # Convert to string
            except:
                raise TypeError(f"Cannot convert object to HDF5-compatible type: {data}")
        else:
            return data

    def save_config(self, sys_config, expt_cfg):
        sys_config = self.convert_for_hdf5(sys_config)
        expt_cfg = self.convert_for_hdf5(expt_cfg)

        #make the folder for today if it doesnt exist already, because we didnt save any of the previous plots
        self.create_folder_if_not_exists(self.outerFolder_expt)
        outerFolder_expt = os.path.join(self.outerFolder_expt, "Data_h5")
        self.create_folder_if_not_exists(outerFolder_expt)
        outerFolder_expt = os.path.join(outerFolder_expt, "configs")
        self.create_folder_if_not_exists(outerFolder_expt)

        with h5py.File(os.path.join(outerFolder_expt, "sys_config.h5") , "w") as hf:
            for key, value in sys_config.items():
                hf.create_dataset(key, data=str(value))

        with h5py.File(os.path.join(outerFolder_expt, "expt_cfg.h5") , "w") as hf:
            for key, value in expt_cfg.items():
                hf.create_dataset(key, data=str(value))

    def load_config(self, filename = 'sys_config.h5'):
        filepath = os.path.join(self.outerFolder_expt, "Data_h5")
        filepath = os.path.join(filepath, 'configs')
        filepath = os.path.join(filepath, filename)
        loaded_config = {}
        with h5py.File(filepath, "r") as hf:
            for key in hf.keys():
                loaded_config[key] = hf[key][()]  # The [()] gets the data as a NumPy array or scalar
        return loaded_config


