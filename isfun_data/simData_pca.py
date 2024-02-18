import os
import pickle
import pkg_resources

def pca_data():
    simData_pca_path = pkg_resources.resource_filename(__name__, 'simData_pca.py')
    script_dir = os.path.dirname(simData_pca_path)
    pkl_file_path = os.path.join(script_dir, 'pca_dict.pkl')
    with open(pkl_file_path, 'rb') as f:
        pca_dict = pickle.load(f)
    return pca_dict
