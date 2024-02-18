import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.preprocessing import scale
from numpy.linalg import svd
from numpy.linalg import solve
from scipy.optimize import root_scalar
from numpy.linalg import lstsq

def meta_spls(x, y, L, mu1, eps=1e-04, kappa=0.05, scale_x=True, scale_y=True, 
              maxstep=50, trace=False):
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
    ip = np.array(range(1, p+1))
    iq = np.array(range(1, q+1))
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
    def ro(x, mu, alpha):
        def f(x):
            return mu * np.where(x/(mu * alpha) < 1, 1 - x/(mu * alpha), 0)
        if np.isscalar(x):
            r = quad(f, 0, x)[0]
        else:
            r = np.array([quad(f, 0, xi)[0] for xi in x])
        return r
    def ro_d1st(x, mu, alpha):
        r = mu * (1 > x/(mu * alpha)) * (1 - x/(mu * alpha))
        return r
    def fun_1(l):
        z_l = np.dot(np.transpose(x[l]), y[l])
        return z_l
    zz = [fun_1(l) for l in range(L)]
    z = np.zeros((p, q))
    for l in range(L):
        z += zz[l]
    z /= np.sum(nl)
    c = svd(np.dot(z, z.T), full_matrices=False)[0][:, :1]
    a = c
    iter = 1
    dis = 10
    kappa2 = (1 - kappa)/(1 - 2 * kappa)
    if trace:
        print("The variables that join the set of selected variables at each step:")
    while dis > eps and iter <= maxstep:
        c_old = c.copy()
        a_old = a.copy()
        m = np.dot(z, z.T) / q
        def h(lambda_):
            alpha = solve(m + lambda_ * np.eye(p), np.dot(m, c))
            obj = np.dot(alpha.T, alpha) - 1/kappa2**2
            return obj
        while h(1e-04) * h(1e+12) > 0:
            m = 2 * m
            c = 2 * c
        lambdas = root_scalar(h, method='brentq', bracket=[1e-04, 1e+12]).root
        a = kappa2 * solve(m + lambdas * np.eye(p), np.dot(m, c))
        s = np.array(np.dot(a.T, np.dot(z, z.T)) / np.sum(nl)).reshape(-1)
        ro_d = ro_d1st(s, mu1, 6)
        fun_c = lambda j: np.sign(s[j]) * (abs(s[j]) > ro_d[j]) * (abs(s[j]) - ro_d[j])
        c = np.array([fun_c(j) for j in range(p)])
        c_norm = np.sqrt(np.sum(np.array(c)**2))
        c_norm = 1e-04 if c_norm == 0 else c_norm
        c = c/c_norm
        dis = np.sqrt(np.sum((c - c_old)**2))/np.sqrt(np.sum(c_old**2))
        what = c.copy()
        what_cut = np.where(np.abs(what) > 1e-04, what, 0)
        what_cut_norm = np.sqrt(np.sum(what_cut**2, axis=0))
        what_cut_norm = 1e-04 if what_cut_norm == 0 else what_cut_norm
        what_dir = (what_cut.T / np.expand_dims(what_cut_norm, axis=-1)).T
        what_cut = np.where(np.abs(what_dir) > 1e-04, what_dir, 0)
        if np.sum(np.abs(c) <= 1e-04) == p:
            print("The value of mu1 is too large")
            break
        if trace:
            new2A = ip[what_cut != 0]
            print("\n--------------------\n----- Step {} -----\n--------------------".format(iter))
            new2A_l = new2A
            if len(new2A_l) <= 10:
                print("DataSet {}:".format(l+1))
                print(', '.join(['X{}'.format(i) for i in new2A_l]))
            else:
                print("DataSet {}:".format(l+1))
                nlines = int(np.ceil(len(new2A_l) / 10))
                for i in range(nlines - 1):
                    print(', '.join(['X{}'.format(i) for i in new2A_l[(10 * i):(10 * (i + 1))]]))
                print(', '.join(['X{}'.format(i) for i in new2A_l[(10 * (nlines - 1)):len(new2A_l)]]))
        iter += 1
    what = what_cut
    new2A = ip[what != 0]
    betahat = np.zeros((p, q * L))
    def fun_fit(l):
        x_l = x[l]
        w_l = np.matrix(what).reshape(-1, 1)
        t_l = np.dot(x_l, w_l)
        if np.sum(w_l == 0) != p:
            y_l = y[l]
            fit_l = lstsq(t_l, y_l, rcond=None)[0]
            betahat_l = np.dot(w_l, fit_l).reshape(p, q)
        else:
            betahat_l = np.zeros((p, q))
        if x[l].shape[1] > 1:
            betahat_l = pd.DataFrame(betahat_l, index=np.arange(1, p+1))
        if q > 1 and y[l].shape[1] > 1:
            betahat_l.columns = np.arange(1, q+1)
        return betahat_l
    betahat = [fun_fit(l) for l in range(L)]
    dictname = [f'Dataset {l+1}' for l in range(L)]
    betahat_dict = {}
    meanx_dict = {}
    meany_dict = {}
    normx_dict = {}
    normy_dict = {}
    x_dict = {}
    y_dict = {}    
    for l in range(L):
        betahat_dict.update({dictname[l]:betahat[l]})
        meanx_dict.update({dictname[l]:meanx[l]})
        meany_dict.update({dictname[l]:meany[l]})
        normx_dict.update({dictname[l]:normx[l]})
        normy_dict.update({dictname[l]:normy[l]})        
        x_dict.update({dictname[l]:x[l]})
        y_dict.update({dictname[l]:y[l]}) 
    class meta_spls:
        def __init__(self, data):
            self.x = data['x']
            self.y = data['y']
            self.betahat = data['betahat']
            self.loading = data['loading']
            self.variable = data['variable']
            self.meanx = data['meanx']
            self.normx = data['normx']
            self.meany = data['meany']
            self.normy = data['normy']
            self.mu1 = data['mu1']
    object_dict = {
        'x': x_dict, 
        'y': y_dict, 
        'betahat': betahat_dict, 
        'loading': what, 
        'variable': new2A,
        'meanx': meanx_dict, 
        'normx': normx_dict, 
        'meany': meany_dict, 
        'normy': normy_dict, 
        'mu1': mu1
    }
    object = meta_spls(object_dict)
    return object