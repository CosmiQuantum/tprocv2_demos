import datetime
import numpy as np
import h5py
import os
np.set_printoptions(threshold=1000000000000000)

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
        ## Check that name/value are alright
        if (type(name)!=type("")) or (value is None):
            raise ValueError("name must be a string and value cannot be NoneType")
            return -1

        ## Check that we aren't trying to save a dictionary directly
        if isinstance(value, dict):
            raise ValueError("Cannot create a dataset directly from a dictionary.")
            return -1

        ## Get the shape of the data
        datashape = np.shape(value)
        testval   = value

        ## This is a scalar value
        if len(datashape)==0:
            ## Save the numpy array of the data we want to write to h5
            datatype  = type(testval)
            writedata = np.array([value]).astype(datatype)
        
        ## This is a list/tuple/np.array (this is mutually exclusive so elif not needed)
        if len(datashape)>0:
            ## Note that casting a list of lists, or a list of tuples to an np array just works
            ## and the resulting type is np.ndarray
            ## Also if you have an np.ndarray and you re-cast it to an np.ndarray, it does nothing
            ## Now we will loop over the shape of the data to pull out the first scalar value, for which
            ## we will check its type
            for i in np.arange(len(datashape)):
                testval = testval[0]

            ## Get the type of scalar data and create a numpy array from the input value
            ## forcing it to be the correct datatype
            datatype  = type(testval)

            if datatype == datetime.datetime:
                value = [time.mktime(vdt.timetuple()) for vdt in value]
                datatype = type(value[0])
            elif datatype == np.datetime64:
                value = [vdt64.astype('datetime64[s]').astype('int64') for vdt64 in value]
                datatype = type(value[0])
            elif data_type == type(None):
                raise ValueError("The scalar data type is NoneType, skipping the write.")
                return -1

            ## Save the numpy array of the data we want to write to h5
            writedata = np.array(value).astype(datatype)

        ## Show the user a warning if for some reason the data is being saved as a string
        if datatype == type(""): 
             print("WARNING: Writing data for "+name+" as a string. Check that this is intended!")

        ## Now try to write the data
        try:
            group.create_dataset(name, data=writedata)
        except Exception as e:
            print("Error: ",e)
            return -1

        return writedata

    def save_to_h5(self, data_type):
        self.outerFolder_expt = os.path.join(self.outerFolder_expt, "Data_h5", f"{data_type}_ge")
        self.create_folder_if_not_exists(self.outerFolder_expt)
        formatted_datetime =  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        h5_filename = os.path.join(self.outerFolder_expt, f"{formatted_datetime}_" + f"{data_type}_results_batch_{self.batch_num}_" + f"Num_per_batch{self.save_r}.h5")
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

        data = {data_type: {}}  # Initialize the main dictionary with the data_type.

        with h5py.File(self.outerFolder_expt, 'r') as f:
            for qubit_group in f.keys():
                qubit_index = int(qubit_group[1:]) - 1
                qubit_data = {}
                group = f[qubit_group]

                for dataset_name in group.keys():
                    # Attempt to map HDF5 keys to the target dictionaries' keys.
                    if data_type == 'Res':
                        target_keys = {'Dates': 'Dates', 'freq_pts': 'freq_pts', 'freq_center': 'freq_center',
                                       'Amps': 'Amps', 'Found Freqs': 'Found Freqs', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                    elif data_type == 'QSpec':
                        target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Frequencies': 'Frequencies',
                                       'I Fit': 'I Fit', 'Q Fit': 'Q Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                    elif data_type == 'Rabi':
                        target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Gains': 'Gains', 'Fit': 'Fit',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num'}
                    elif data_type == 'SS':
                        target_keys = {'Fidelity': 'Fidelity', 'Angle': 'Angle', 'Dates': 'Dates', 'I_g': 'I_g', 'Q_g': 'Q_g', 'I_e': 'I_e', 'Q_e': 'Q_e',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num'}
                    elif data_type == 'T1':
                        target_keys = {'T1': 'T1', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                    elif data_type == 'T2':
                        target_keys = {'T2': 'T2', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                    elif data_type == 'T2E':
                        target_keys = {'T2E': 'T2E', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                    else:
                        raise ValueError(f"Unsupported data_type: {data_type}")

                    try:
                        mapped_key = target_keys[dataset_name]  # Map HDF5 key to target key.
                        qubit_data[mapped_key] = [group[dataset_name][
                                                      ()]] * save_r  # Expand to match the desired length.

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


