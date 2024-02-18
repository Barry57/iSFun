import os
import pickle
import pkg_resources

def pls_data():
    simData_pls_path = pkg_resources.resource_filename(__name__, 'simData_pls.py')
    script_dir = os.path.dirname(simData_pls_path)
    pkl_file_path = os.path.join(script_dir, 'pls_dict.pkl')
    with open(pkl_file_path, 'rb') as f:
        pls_dict = pickle.load(f)
    return pls_dict