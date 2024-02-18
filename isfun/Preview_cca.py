import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

def preview_cca(x, y, L, scale_x=True, scale_y=True):
    if not isinstance(x, list):
        raise TypeError("x should be of list type.")
    if not isinstance(y, list):
        raise TypeError("y should be of list type.")
    x = [np.array(i) for i in x]
    y = [np.array(i) for i in y]  
    nl = [i.shape[0] for i in x]
    pl = [i.shape[1] for i in x]    
    ql = [i.shape[1] for i in y]  
    p = list(set(pl))
    q = list(set(ql))   
    if len(p) > 1:
        raise ValueError("The dimension of data x should be consistent among different datasets.")
    if len(q) > 1:
        raise ValueError("The dimension of data y should be consistent among different datasets.")
    if len(p) == 1:
        p = p[0]
    if len(q) == 1:
        q = q[0]
    meanx = [np.dot(np.ones((1, nl[l])), x[l])/nl[l] for l in range(L)]
    meany = [np.dot(np.ones((1, nl[l])), y[l])/nl[l] for l in range(L)]
    for l in range(L):
        x[l] = scale(x[l], with_std=False)
        y[l] = scale(y[l], with_std=False)
    def x_scale(l):
        one = np.ones((1, nl[l]))
        normx = np.sqrt(np.dot(one, x[l]**2) / (nl[l] - 1))
        if np.any(normx < np.finfo(float).eps):
            raise ValueError("Some of the columns of the predictor matrix have zero variance.")
        return normx
    def y_scale(l):
        one = np.ones((1, nl[l]))
        normy = np.sqrt(np.dot(one, y[l]**2) / (nl[l] - 1))
        if np.any(normy < np.finfo(float).eps):
            raise ValueError("Some of the columns of the response matrix have zero variance.")
        return normy
    if scale_x:
        normx = [x_scale(i) for i in range(L)]
    else:
        normx = [[1]*p[0]]*L   
    if scale_y:
        normy = [y_scale(i) for i in range(L)]
    else:
        normy = [[1]*q[0]]*L
    if scale_x:
        x = [(x[i] - np.mean(x[i], axis=0)) / normx[i] for i in range(L)]
    if scale_y:
        y = [(y[i] - np.mean(y[i], axis=0)) / normy[i] for i in range(L)]
    def fun_1(l):
        Z_l = svds(np.dot(x[l].T, y[l]), k=1)
        u_l = Z_l[0]
        return u_l
    U = np.hstack(np.array([fun_1(l) for l in range(L)]))
    def fun_2(l):
        Z_l = svds(np.dot(x[l].T, y[l]), k=1)
        v_l = Z_l[2]
        return v_l
    V = np.vstack(np.array([fun_2(l) for l in range(L)])).T
    what_u = U
    what_v = V
    dictname = [f'Dataset {l+1}' for l in range(L)]
    meanx_dict = {}
    meany_dict = {}
    normx_dict = {}
    normy_dict = {}
    x_dict = {}
    y_dict = {}    
    for l in range(L):
        meanx_dict.update({dictname[l]:meanx[l]})
        meany_dict.update({dictname[l]:meany[l]})
        normx_dict.update({dictname[l]:normx[l]})
        normy_dict.update({dictname[l]:normy[l]})        
        x_dict.update({dictname[l]:x[l]})
        y_dict.update({dictname[l]:y[l]})
    what_u_df = pd.DataFrame(what_u, index = [str(i+1) for i in range(p)], columns = dictname)
    what_v_df = pd.DataFrame(what_v, index = [str(i+1) for i in range(p)], columns = dictname)
    def plot_loading(order, U, V, p, q):
        for l in order:
            fig, axs = plt.subplots(1, 2)
            axs[0].scatter(np.arange(1, p+1), U[:, l-1])
            axs[0].set_title(f'Dataset {l}\nThe first canonical vector u')
            axs[0].set_xlabel('Dimension')
            axs[0].set_ylabel('Value')   
            axs[1].scatter(np.arange(1, q+1), V[:, l-1])
            axs[1].set_title(f'Dataset {l}\nThe first canonical vector v')
            axs[1].set_xlabel('Dimension')
            axs[1].set_ylabel('Value') 
            plt.show()
    plot_loading(order = np.arange(1, L+1), U=U, V=V, p=p, q=q)
    class preview_cca:
        def __init__(self, data):
            self.x = data['x']
            self.y = data['y']
            self.loading_x = data['loading.x']
            self.loading_y = data['loading.y']
            self.meanx = data['meanx']
            self.normx = data['normx']
            self.meany = data['meany']
            self.normy = data['normy']
    object_dict = {
        'x': x_dict, 
        'y': y_dict, 
        'loading.x': what_u_df, 
        'loading.y': what_v_df, 
        'meanx': meanx_dict, 
        'normx': normx_dict, 
        'meany': meany_dict, 
        'normy': normy_dict, 
    }
    object = preview_cca(object_dict)
    return object
