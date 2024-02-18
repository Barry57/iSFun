import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns

def preview_pca(x, L, scale_x=True):
    if not isinstance(x, list):
        raise TypeError("x should be of list type.")
    x = [np.array(i) for i in x]
    nl = [i.shape[0] for i in x]
    pl = [i.shape[1] for i in x]     
    p = list(set(pl)) 
    if len(p) > 1:
        raise ValueError("The dimension of data x should be consistent among different datasets.")
    if len(p) == 1:
        p = p[0]
    meanx = [np.dot(np.ones((1, nl[l])), x[l])/nl[l] for l in range(L)]
    for l in range(L):
        x[l] = scale(x[l], with_std=False)
    def x_scale(l):
        one = np.ones((1, nl[l]))
        normx = np.sqrt(np.dot(one, x[l]**2) / (nl[l] - 1))
        if np.any(normx < np.finfo(float).eps):
            raise ValueError("Some of the columns of the predictor matrix have zero variance.")
        return normx
    if scale_x:
        normx = [x_scale(i) for i in range(L)]
    else:
        normx = [[1]*p[0]]*L   
    if scale_x:
        x = [(x[i] - np.mean(x[i], axis=0)) / normx[i] for i in range(L)]
    what = np.zeros((p, L))
    def fun_1(l):
         u_l, s_l, v_l = svds(x[l], k=1)
         return v_l.T
    U = np.hstack(np.array([fun_1(l) for l in range(L)]))
    what = U
    eigenvalue = [what[:, l].T.dot(np.cov(x[l], rowvar = False)).dot(what[:, l]) for l in range(L)]
    comp = [x[l].dot(what[:, l]) for l in range(L)]
    dictname = [f'Dataset {l+1}' for l in range(L)]
    meanx_dict = {}
    normx_dict = {}
    x_dict = {}
    eigenvalue_dict = {}
    comp_dict = {}
    for l in range(L):
        meanx_dict.update({dictname[l]:meanx[l]})
        normx_dict.update({dictname[l]:normx[l]})      
        x_dict.update({dictname[l]:x[l]})
        eigenvalue_dict.update({dictname[l]:eigenvalue[l]})
        comp_dict.update({dictname[l]:comp[l]})
    what_df = pd.DataFrame(what, index = [str(i+1) for i in range(p)], columns = dictname)
    dictname = [f'Dataset {l+1}' for l in range(L)]
    meanx_dict = {}
    normx_dict = {}
    x_dict = {}   
    for l in range(L):
        meanx_dict.update({dictname[l]:meanx[l]})
        normx_dict.update({dictname[l]:normx[l]})       
        x_dict.update({dictname[l]:x[l]})
    what_df = pd.DataFrame(what, index = [str(i+1) for i in range(p)], columns = dictname)
    for l in range(L):
        plt.figure()
        plt.scatter(x = np.arange(1, p+1), y = what_df.iloc[:, l])
        plt.title(f"The first principal component of dataset {l+1}")
        plt.xlabel("Dimension")
        plt.ylabel("Value")
    heat = [np.cov(i) for i in x]
    for l in range(L):
        plt.figure()
        sns.heatmap(heat[l], cmap = 'Reds', xticklabels=False, yticklabels=False)
        plt.xlabel(f"Dataset {l+1}: X")
        plt.ylabel("Y")
        plt.title("Heatmap of covariance")
    class preview_pca:
        def __init__(self, data):
            self.x = data['x']
            self.eigenvalue = data['eigenvalue']
            self.eigenvector = data['eigenvector']
            self.component = data['component']
            self.meanx = data['meanx']
            self.normx = data['normx']
    object_dict = {
        'x': x_dict, 
        'eigenvalue': eigenvalue_dict, 
        'eigenvector': what_df, 
        'component': comp_dict, 
        'meanx': meanx_dict,
        'normx': normx_dict, 
    }
    object = preview_pca(object_dict)
    return object