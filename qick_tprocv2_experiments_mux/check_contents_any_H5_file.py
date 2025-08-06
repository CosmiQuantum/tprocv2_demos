import h5py
import numpy as np

def print_h5_keys_verbose(h5_path):
    with h5py.File(h5_path, 'r') as f:
        def visit_func(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"[Group]   {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"[Dataset] {name} - shape: {obj.shape}, dtype: {obj.dtype}")
        f.visititems(visit_func)

#--------------------------------------------------
h5_path = "/exp/cosmiq/data/QUIET/QICK_data/run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20/2024-12-10/study_data/Data_h5/ss_ge/2024-12-10_19-17-36_SS_results_batch_1_Num_per_batch1.h5"
print_h5_keys_verbose(h5_path)

with h5py.File(h5_path, 'r') as f:
    dates = f['Q1']['Dates'][()]
    print("Dates content:", dates)
    print("Is NaN?", np.isnan(dates[0]))