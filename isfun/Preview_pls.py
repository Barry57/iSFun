import numpy as np
from sklearn.preprocessing import scale
from numpy.linalg import svd
import matplotlib.pyplot as plt

def preview_pls(x, y, L, scale_x=True, scale_y=True):
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
        z_l = np.dot(np.transpose(x[l]), y[l])
        z_l = z_l / nl[l]
        return z_l
    z = np.hstack(np.array([fun_1(l) for l in range(L)]))
    def fun_c(l):
        return svd(np.dot(z[:, (l * q):(l + 1) * q], z[:, (l * q):(l + 1) * q].T), full_matrices=False)[0][:, 0]
    c = np.array([fun_c(l) for l in range(L)]).T
    what = c
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
    for l in range(L):
        plt.figure()
        plt.scatter(range(1, p+1), what[:, l])
        plt.title('Dataset {}\nThe first direction vector'.format(l+1))
        plt.xlabel('Dimension')
        plt.ylabel('Value')
        plt.show()
    class preview_pls:
        def __init__(self, data):
            self.x = data['x']
            self.y = data['y']
            self.loading_x = data['loading']
            self.meanx = data['meanx']
            self.normx = data['normx']
            self.meany = data['meany']
            self.normy = data['normy']
    object_dict = {
        'x': x_dict, 
        'y': y_dict, 
        'loading': what, 
        'meanx': meanx_dict, 
        'normx': normx_dict, 
        'meany': meany_dict, 
        'normy': normy_dict, 
    }
    object = preview_pls(object_dict)
    return object