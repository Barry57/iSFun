import os
import pickle
import pkg_resources

def cca_data():
    simData_cca_path = pkg_resources.resource_filename(__name__, 'simData_cca.py')
    script_dir = os.path.dirname(simData_cca_path)
    pkl_file_path = os.path.join(script_dir, 'cca_dict.pkl')
    with open(pkl_file_path, 'rb') as f:
        cca_dict = pickle.load(f)
    return cca_dict
