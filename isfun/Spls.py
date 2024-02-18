import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.preprocessing import scale
from numpy.linalg import solve
from numpy.linalg import svd
from scipy.optimize import root_scalar
from numpy.linalg import lstsq

def spls(x, y, mu1, eps=1e-04, kappa=0.05, scale_x=True, scale_y=True, maxstep=50, trace=False):
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    nx, p = x.shape
    ny, q = y.shape
    if nx != ny:
        raise ValueError("The rows of data x and data y should be consistent.") 
    n = nx
    ip = np.arange(1,p+1)
    one = np.ones((1, n))
    meany = np.dot(one, y) / n
    y = scale(y, with_std=False)
    meanx = np.dot(one, x) / n
    x = scale(x, with_std=False)
    if scale_x:
        normx = np.sqrt(np.dot(one, x**2) / (n - 1))
        if np.any(normx < np.finfo(float).eps):
            raise ValueError("Some of the columns of the predictor matrix have zero variance.")
        x = scale(x, with_std=False) / normx
    else:
        normx = np.ones(p)
    if scale_y:
        normy = np.sqrt(np.dot(one, y**2) / (n - 1))
        if np.any(normy < np.finfo(float).eps):
            raise ValueError("Some of the columns of the response matrix have zero variance.")
        y = scale(y, with_std=False) / normy
    else:
        normy = np.ones(q)
    what = np.zeros((p, 1))
    if trace:
        print("The variables that join the set of selected variables at each step:")
    z = np.dot(x.T, y)
    z = z / n
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
    c = svd(np.dot(z,z.T), full_matrices=False)[0][:, 0]
    a = c
    iter = 1
    dis = 10
    kappa2 = (1 - kappa) / (1 - 2 * kappa)
    loading_trace = np.zeros((p, maxstep))
    while dis > eps and iter <= maxstep:
        c_old = c
        a_old = a
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
        s = np.array(np.dot(a.T, np.dot(z, z.T)) / np.sum(n)).reshape(-1)
        ro_d = ro_d1st(s, mu1, 6)
        fun_c = lambda j: np.sign(s[j]) * (abs(s[j]) > ro_d[j]) * (abs(s[j]) - ro_d[j])
        c = np.array([fun_c(j) for j in range(p)])
        c_norm = np.sqrt(sum(np.array(c)**2))
        c_norm = 1e-04 if c_norm == 0 else c_norm
        c = c / c_norm
        dis = np.sqrt(sum((c - c_old)**2)) / np.sqrt(sum(np.array(c_old)**2))
        what = c
        what_cut = np.where(abs(what) > 1e-04, what, 0)
        what_cut_norm = np.sqrt(sum(what_cut**2))
        what_cut_norm = 1e-04 if what_cut_norm == 0 else what_cut_norm
        what_dir = (what_cut.T / np.expand_dims(what_cut_norm, axis=-1)).T
        what_cut = np.where(abs(what_dir) > 1e-04, what_dir, 0)
        loading_trace[:, iter-1] = what_cut
        if sum(abs(c) <= 1e-04) == p:
            print("The value of mu1 is too large")
            break
        if trace:
            new2A = ip[what_cut != 0]
            print("\n--------------------\n----- Step {} -----\n--------------------".format(iter))
            if len(new2A) <= 10:
                print(', '.join(['X{}'.format(i) for i in new2A]))
            else:
                nlines = int(np.ceil(len(new2A) / 10))
                for i in range(nlines - 1):
                    print(', '.join(['X{}'.format(i) for i in new2A[(10 * i):(10 * (i + 1))]]))
                print(', '.join(['X{}'.format(i) for i in new2A[(10 * (nlines - 1)):len(new2A)]]))
        iter += 1
    loading_trace = loading_trace[:, :iter-1]
    what = what_cut
    new2A = ip[what != 0]
    betahat = np.zeros((p, q))
    w = what
    t_hat = np.dot(x, w)
    if sum(w == 0) != p:
        t_hat = t_hat.reshape(-1, 1)
        betahat = np.dot(np.expand_dims(w, axis=-1), lstsq(t_hat, y, rcond=None)[0]).reshape(p, q)
    else:
        betahat = np.zeros((p, q))
    if x.shape[1] > 1:
        betahat = pd.DataFrame(betahat, index=range(1, p + 1))
    if q > 1 and y.shape[1] > 1:
        betahat.columns = range(1, q + 1)
    class spls:
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
        'x': x, 
        'y': y, 
        'betahat': betahat, 
        'loading': what, 
        'variable': new2A,
        'meanx': meanx, 
        'normx': normx, 
        'meany': meany, 
        'normy': normy, 
        'mu1': mu1
    }
    object = spls(object_dict)
    return object
