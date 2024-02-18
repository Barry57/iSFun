import numpy as np
from scipy.integrate import quad
from sklearn.preprocessing import scale
from scipy.sparse.linalg import svds

def scca(x, y, mu1, mu2, eps=1e-04, scale_x=True, scale_y=True, maxstep=50, trace=False):
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    nx, p = x.shape
    ny, q = y.shape
    if nx != ny:
        raise ValueError("The rows of data x and data y should be consistent.") 
    n = nx
    ip = np.arange(1,p+1)
    iq = np.arange(1,q+1)
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
    U, s, Vt = svds(np.dot(x.T, y), k=1)
    u = U
    v = Vt.T
    iter = 1
    dis_u = 10
    dis_v = 10
    if trace:
        print("The variables that join the set of selected variables at each step:")
    while dis_u > eps and dis_v > eps and iter <= maxstep:
        u_old = u.copy()
        v_old = v.copy()
        s = np.array(np.dot(np.dot(v.T, y.T), x) / n).reshape(-1)
        ro_d = np.array([ro_d1st(s[j], mu1, 6) for j in range(p)])
        fun_c = lambda j: np.sign(s[j]) * (abs(s[j]) > ro_d[j]) * (abs(s[j]) - ro_d[j])
        u = np.array([fun_c(j) for j in range(p)])
        u_norm = np.sqrt(np.sum(u**2))
        u_norm = 1e-04 if u_norm == 0 else u_norm
        u = u / u_norm
        dis_u = np.sqrt(np.sum((u - u_old)**2)) / np.sqrt(np.sum(u_old**2))
        what = u
        what_cut = np.where(abs(what) > 1e-04, what, 0)
        what_cut_norm = np.sqrt(np.sum(what_cut**2))
        what_cut_norm = 1e-04 if what_cut_norm == 0 else what_cut_norm
        what_dir = what_cut / what_cut_norm
        what_cut = np.where(abs(what_dir) > 1e-04, what_dir, 0)
        s = np.dot(np.dot(u.T, x.T), y) / n
        ro_d = np.array([ro_d1st(s[j], mu2, 6) for j in range(q)])
        fun_c = lambda j: np.sign(s[j]) * (abs(s[j]) > ro_d[j]) * (abs(s[j]) - ro_d[j])
        v = np.array([fun_c(j) for j in range(q)])
        v_norm = np.sqrt(np.sum(v**2))
        v_norm = 1e-04 if v_norm == 0 else v_norm
        v = v / v_norm
        dis_v = np.sqrt(np.sum((v - v_old)**2)) / np.sqrt(np.sum(v_old**2))
        what_v = v
        what_cut_v = np.where(abs(what_v) > 1e-04, what_v, 0)
        what_cut_norm = np.sqrt(np.sum(what_cut_v**2))
        what_cut_norm = 1e-04 if what_cut_norm == 0 else what_cut_norm
        what_dir = what_cut_v / what_cut_norm
        what_cut_v = np.where(abs(what_dir) > 1e-04, what_dir, 0)
        dis_u = np.sqrt(np.sum((u - u_old)**2)) / np.sqrt(np.sum(u_old**2))
        dis_v = np.sqrt(np.sum((v - v_old)**2)) / np.sqrt(np.sum(v_old**2))
        if np.sum(abs(v) <= 1e-04) == q and np.sum(abs(u) <= 1e-04) == p:
            print("Stop! The values of mu1 and mu2 are too large")
            break
        if np.sum(abs(v) <= 1e-04) != q and np.sum(abs(u) <= 1e-04) == p:
            print("Stop! The value of mu1 is too large")
            break
        if np.sum(abs(v) <= 1e-04) == q and np.sum(abs(u) <= 1e-04) != p:
            print("Stop! The value of mu2 is too large")
            break
        if trace:
            new2A_u = ip[what_cut != 0]
            new2A_v = iq[what_cut_v != 0]
            print("\n--------------------\n----- Step {} -----\n--------------------".format(iter))
            new2A_l = new2A_u
            if len(new2A_l) <= 10:
                print(', '.join(['X{}'.format(i) for i in new2A_l]))
            else:
                nlines = int(np.ceil(len(new2A_l) / 10))
                for i in range(nlines - 1):
                    print(', '.join(['X{}'.format(i) for i in new2A_l[(10 * i):(10 * (i + 1))]]))
                print(', '.join(['X{}'.format(i) for i in new2A_l[(10 * (nlines - 1)):len(new2A_l)]]))
            new2A_l = new2A_v
            if len(new2A_l) <= 10:
                print(', '.join(['Y{}'.format(i) for i in new2A_l]))
            else:
                nlines = int(np.ceil(len(new2A_l) / 10))
                for i in range(nlines - 1):
                    print(', '.join(['Y{}'.format(i) for i in new2A_l[(10 * i):(10 * (i + 1))]]))
                print(', '.join(['Y{}'.format(i) for i in new2A_l[(10 * (nlines - 1)):len(new2A_l)]]))
        iter += 1
    what = what_cut
    what_v = what_cut_v
    new2A = ip[what != 0]
    new2A_v = iq[what_v != 0]  
    class scca:
        def __init__(self, data):
            self.x = data['x']
            self.y = data['y']
            self.loading_x = data['loading.x']
            self.loading_y = data['loading.y']
            self.variable_x = data['variable.x']
            self.variable_y = data['variable.y']
            self.meanx = data['meanx']
            self.normx = data['normx']
            self.meany = data['meany']
            self.normy = data['normy']
            self.mu1 = data['mu1']
            self.mu2 = data['mu2']
    object_dict = {
        'x': x, 
        'y': y, 
        'loading.x': what, 
        'loading.y': what_v, 
        'variable.x': new2A, 
        'variable.y': new2A_v, 
        'meanx': meanx, 
        'normx': normx, 
        'meany': meany, 
        'normy': normy, 
        'mu1': mu1, 
        'mu2': mu2, 
    }
    object = scca(object_dict)
    return object
